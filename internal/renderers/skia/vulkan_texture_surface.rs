//! Skia Vulkan renderer for rendering Slint UI to an existing Vulkan image.

use std::cell::RefCell;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;

use i_slint_core::api::PhysicalSize as PhysicalWindowSize;
use i_slint_core::graphics::rendering_metrics_collector::RenderingMetricsCollector;
use i_slint_core::item_rendering::ItemCache;
use i_slint_core::lengths::{LogicalLength, LogicalPoint, LogicalRect, LogicalSize};
use i_slint_core::platform::PlatformError;
use i_slint_core::renderer::RendererSealed;
use i_slint_core::textlayout::sharedparley;
use i_slint_core::window::{WindowAdapter, WindowInner};

use vulkano::device::{Device, Queue};
use vulkano::format::Format as VulkanoFormat;
use vulkano::image::{Image, ImageLayout as VulkanoImageLayout};
use vulkano::{Handle, VulkanObject};

pub struct SkiaVulkanTextureRenderer {
    gr_context: RefCell<skia_safe::gpu::DirectContext>,
    device: Arc<Device>,
    queue: Arc<Queue>,

    // Rendering state - same as SkiaRenderer
    maybe_window_adapter: RefCell<Option<std::rc::Weak<dyn WindowAdapter>>>,
    image_cache: ItemCache<Option<skia_safe::Image>>,
    layer_cache: ItemCache<
        Option<(
            i_slint_core::graphics::euclid::Vector2D<f32, i_slint_core::lengths::PhysicalPx>,
            skia_safe::Image,
        )>,
    >,
    path_cache: ItemCache<
        Option<(
            i_slint_core::graphics::euclid::Vector2D<f32, i_slint_core::lengths::PhysicalPx>,
            skia_safe::Path,
        )>,
    >,
    text_layout_cache: sharedparley::TextLayoutCache,
    rendering_metrics_collector: RefCell<Option<Rc<RenderingMetricsCollector>>>,

    // Target image info (set before each render)
    target_image: RefCell<Option<Arc<Image>>>,
    target_size: RefCell<PhysicalWindowSize>,
}

impl SkiaVulkanTextureRenderer {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Result<Self, PlatformError> {
        let gr_context = {
            let physical_device = device.physical_device();
            let instance = physical_device.instance();
            let library = instance.library();

            // Log diagnostic information for debugging context creation failures
            let props = physical_device.properties();
            let extensions = physical_device.supported_extensions();
            let features = physical_device.supported_features();

            eprintln!("[Skia] Creating Vulkan context with:");
            eprintln!("  Device: {} ({:?})", props.device_name, props.device_type);
            eprintln!(
                "  API version: {}.{}.{}",
                props.api_version.major, props.api_version.minor, props.api_version.patch
            );
            eprintln!("  Driver version: {}", props.driver_version);
            eprintln!(
                "  Queue family: {}, index: {}",
                queue.queue_family_index(),
                queue.queue_index()
            );

            // Check queue family capabilities
            let queue_family_props =
                &physical_device.queue_family_properties()[queue.queue_family_index() as usize];
            eprintln!("  Queue flags: {:?}", queue_family_props.queue_flags);

            // Extensions that Skia's Vulkan backend may require
            eprintln!("[Skia] Checking critical extensions:");
            eprintln!("  VK_KHR_swapchain: {}", extensions.khr_swapchain);
            eprintln!(
                "  VK_KHR_get_memory_requirements2: {}",
                extensions.khr_get_memory_requirements2
            );
            eprintln!("  VK_KHR_dedicated_allocation: {}", extensions.khr_dedicated_allocation);
            eprintln!("  VK_KHR_external_memory: {}", extensions.khr_external_memory);
            eprintln!(
                "  VK_KHR_sampler_ycbcr_conversion: {}",
                extensions.khr_sampler_ycbcr_conversion
            );

            // Features that Skia may require
            eprintln!("[Skia] Checking critical features:");
            eprintln!("  sampler_anisotropy: {}", features.sampler_anisotropy);
            eprintln!("  shader_clip_distance: {}", features.shader_clip_distance);

            // Check if the device was created with necessary extensions enabled
            let enabled_extensions = device.enabled_extensions();
            eprintln!("[Skia] Extensions enabled on device:");
            eprintln!("  VK_KHR_swapchain: {}", enabled_extensions.khr_swapchain);
            eprintln!(
                "  VK_KHR_get_memory_requirements2: {}",
                enabled_extensions.khr_get_memory_requirements2
            );
            eprintln!(
                "  VK_KHR_dedicated_allocation: {}",
                enabled_extensions.khr_dedicated_allocation
            );

            let get_proc = |of| unsafe {
                let result = match of {
                    skia_safe::gpu::vk::GetProcOf::Instance(instance, name) => {
                        library.get_instance_proc_addr(
                            ash::vk::Instance::from_raw(instance as _),
                            name,
                        )
                    }
                    skia_safe::gpu::vk::GetProcOf::Device(vk_device, name) => {
                        (instance.fns().v1_0.get_device_proc_addr)(
                            ash::vk::Device::from_raw(vk_device as _),
                            name,
                        )
                    }
                };

                match result {
                    Some(f) => f as _,
                    None => core::ptr::null(),
                }
            };

            // Tell Skia which device extensions are actually enabled so it doesn't
            // try to use features that are available but not enabled on the device.
            let enabled_device_exts: Vec<&str> = [
                (enabled_extensions.khr_swapchain, "VK_KHR_swapchain"),
                (enabled_extensions.khr_get_memory_requirements2, "VK_KHR_get_memory_requirements2"),
                (enabled_extensions.khr_dedicated_allocation, "VK_KHR_dedicated_allocation"),
                (enabled_extensions.khr_external_memory, "VK_KHR_external_memory"),
                (enabled_extensions.khr_sampler_ycbcr_conversion, "VK_KHR_sampler_ycbcr_conversion"),
                (enabled_extensions.khr_maintenance1, "VK_KHR_maintenance1"),
                (enabled_extensions.khr_maintenance2, "VK_KHR_maintenance2"),
                (enabled_extensions.khr_maintenance3, "VK_KHR_maintenance3"),
            ].iter().filter(|(enabled, _)| *enabled).map(|(_, name)| *name).collect();

            eprintln!("[Skia] Creating BackendContext with {} device extensions...", enabled_device_exts.len());
            let mut backend_context = unsafe {
                skia_safe::gpu::vk::BackendContext::new_with_extensions(
                    instance.handle().as_raw() as _,
                    physical_device.handle().as_raw() as _,
                    device.handle().as_raw() as _,
                    (queue.handle().as_raw() as _, queue.queue_index() as _),
                    &get_proc,
                    &[],
                    &enabled_device_exts,
                )
            };

            // Cap Skia to Vulkan 1.3 unconditionally. Skia doesn't need 1.4 features
            // for UI overlay rendering, and many drivers (especially AMD) fail when Skia
            // probes for optional 1.4 functions like vkTransitionImageLayout from
            // VK_EXT_host_image_copy.
            // See: https://github.com/rust-skia/rust-skia/issues/513
            let vk_1_3: u32 = (1 << 22) | (3 << 12);
            backend_context.set_max_api_version(vk_1_3);
            eprintln!("[Skia] BackendContext created successfully (capped to Vulkan 1.3)");

            eprintln!("[Skia] Calling make_vulkan (DirectContext creation)...");
            let context = skia_safe::gpu::direct_contexts::make_vulkan(&backend_context, None);

            match context {
                Some(ctx) => {
                    eprintln!("[Skia] DirectContext created successfully!");
                    ctx
                }
                None => {
                    eprintln!("[Skia] ERROR: make_vulkan returned None!");
                    eprintln!("[Skia] This usually indicates one of:");
                    eprintln!("  1. Missing or incompatible Vulkan driver");
                    eprintln!("  2. Required Vulkan extension not available or not enabled");
                    eprintln!("  3. Required Vulkan feature not available or not enabled");
                    eprintln!("  4. Skia internal validation failure");
                    eprintln!(
                        "[Skia] Try updating GPU drivers or check if another app is using the GPU exclusively"
                    );
                    return Err(
                        "Error creating Skia Vulkan context (make_vulkan returned None)".into()
                    );
                }
            }
        };

        Ok(Self {
            gr_context: RefCell::new(gr_context),
            device,
            queue,
            maybe_window_adapter: Default::default(),
            image_cache: Default::default(),
            layer_cache: Default::default(),
            path_cache: Default::default(),
            text_layout_cache: Default::default(),
            rendering_metrics_collector: Default::default(),
            target_image: RefCell::new(None),
            target_size: RefCell::new(PhysicalWindowSize::default()),
        })
    }

    pub fn render_to_vulkan_image(
        &self,
        target_image: &Arc<Image>,
        width: u32,
        height: u32,
    ) -> Result<(), PlatformError> {
        self.render_impl(
            target_image,
            width,
            height,
            skia_safe::gpu::vk::Format::B8G8R8A8_UNORM,
            skia_safe::ColorType::BGRA8888,
            skia_safe::gpu::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            false,
        )
    }

    pub fn render_to_vulkan_image_sync(
        &self,
        target_image: &Arc<Image>,
        width: u32,
        height: u32,
    ) -> Result<(), PlatformError> {
        self.render_impl(
            target_image,
            width,
            height,
            skia_safe::gpu::vk::Format::B8G8R8A8_UNORM,
            skia_safe::ColorType::BGRA8888,
            skia_safe::gpu::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            true,
        )
    }

    pub fn render_to_vulkan_image_with_format(
        &self,
        target_image: &Arc<Image>,
        width: u32,
        height: u32,
        format: VulkanoFormat,
    ) -> Result<(), PlatformError> {
        let (skia_format, color_type) = vulkano_format_to_skia(format)?;
        self.render_impl(
            target_image,
            width,
            height,
            skia_format,
            color_type,
            skia_safe::gpu::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            false,
        )
    }

    pub fn render_to_vulkan_image_with_format_sync(
        &self,
        target_image: &Arc<Image>,
        width: u32,
        height: u32,
        format: VulkanoFormat,
    ) -> Result<(), PlatformError> {
        let (skia_format, color_type) = vulkano_format_to_skia(format)?;
        self.render_impl(
            target_image,
            width,
            height,
            skia_format,
            color_type,
            skia_safe::gpu::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            true,
        )
    }

    pub fn render_to_vulkan_image_with_layout(
        &self,
        target_image: &Arc<Image>,
        width: u32,
        height: u32,
        layout: VulkanoImageLayout,
    ) -> Result<(), PlatformError> {
        let skia_layout = vulkano_layout_to_skia(layout)?;
        self.render_impl(
            target_image,
            width,
            height,
            skia_safe::gpu::vk::Format::B8G8R8A8_UNORM,
            skia_safe::ColorType::BGRA8888,
            skia_layout,
            false,
        )
    }

    /// Core render implementation with all options.
    fn render_impl(
        &self,
        target_image: &Arc<Image>,
        width: u32,
        height: u32,
        vk_format: skia_safe::gpu::vk::Format,
        color_type: skia_safe::ColorType,
        image_layout: skia_safe::gpu::vk::ImageLayout,
        sync_cpu: bool,
    ) -> Result<(), PlatformError> {
        *self.target_image.borrow_mut() = Some(target_image.clone());
        *self.target_size.borrow_mut() = PhysicalWindowSize::new(width, height);

        let result =
            self.render_internal_with_options(vk_format, color_type, image_layout, sync_cpu);

        *self.target_image.borrow_mut() = None;

        result
    }

    fn render_internal_with_options(
        &self,
        vk_format: skia_safe::gpu::vk::Format,
        color_type: skia_safe::ColorType,
        image_layout: skia_safe::gpu::vk::ImageLayout,
        sync_cpu: bool,
    ) -> Result<(), PlatformError> {
        let target_image = self.target_image.borrow();
        let target_image = target_image.as_ref().ok_or("No target image set for rendering")?;
        let size = *self.target_size.borrow();

        let gr_context = &mut self.gr_context.borrow_mut();

        let width: i32 = size.width.try_into().map_err(|_| "Invalid image width")?;
        let height: i32 = size.height.try_into().map_err(|_| "Invalid image height")?;

        let alloc = skia_safe::gpu::vk::Alloc::default();
        let image_info = unsafe {
            skia_safe::gpu::vk::ImageInfo::new(
                target_image.handle().as_raw() as _,
                alloc,
                skia_safe::gpu::vk::ImageTiling::OPTIMAL,
                image_layout,
                vk_format,
                1,
                None,
                None,
                None,
                None,
            )
        };

        let render_target =
            skia_safe::gpu::backend_render_targets::make_vk((width, height), &image_info);

        let mut skia_surface = skia_safe::gpu::surfaces::wrap_backend_render_target(
            gr_context,
            &render_target,
            skia_safe::gpu::SurfaceOrigin::TopLeft,
            color_type,
            None,
            None,
        )
        .ok_or_else(|| "Error creating Skia surface for Vulkan image".to_string())?;

        // Render the Slint UI
        self.render_to_canvas(skia_surface.canvas(), Some(gr_context));

        drop(skia_surface);

        // Submit Skia's command buffer, optionally waiting for GPU completion
        if sync_cpu {
            gr_context.flush_submit_and_sync_cpu();
        } else {
            gr_context.submit(None);
        }

        Ok(())
    }

    fn render_to_canvas(
        &self,
        canvas: &skia_safe::Canvas,
        mut gr_context: Option<&mut skia_safe::gpu::DirectContext>,
    ) {
        let Some(window_adapter) =
            self.maybe_window_adapter.borrow().as_ref().and_then(|w| w.upgrade())
        else {
            return;
        };

        let window = window_adapter.window();
        let window_inner = WindowInner::from_pub(window);

        self.image_cache.clear_cache_if_scale_factor_changed(window);
        self.path_cache.clear_cache_if_scale_factor_changed(window);
        self.text_layout_cache.clear_cache_if_scale_factor_changed(window);

        let mut box_shadow_cache = Default::default();

        let mut skia_item_renderer = crate::itemrenderer::SkiaItemRenderer::new(
            canvas,
            window,
            None, // No surface for texture import (overlay mode)
            &self.image_cache,
            &self.layer_cache,
            &self.path_cache,
            &self.text_layout_cache,
            &mut box_shadow_cache,
        );

        // Clear the canvas to transparent before rendering.
        // This is essential when rendering to a dedicated texture that will be composited later.
        canvas.clear(skia_safe::Color::TRANSPARENT);

        window_inner.draw_contents(|components| {
            for (component, origin) in components {
                if let Some(component) = i_slint_core::item_tree::ItemTreeWeak::upgrade(component) {
                    i_slint_core::item_rendering::render_component_items(
                        &component,
                        &mut skia_item_renderer,
                        *origin,
                        &window_adapter,
                    );
                }
            }
        });

        if let Some(ctx) = gr_context.as_mut() {
            ctx.flush(None);
        }
    }

    /// Returns the Vulkano device.
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns the Vulkano queue.
    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }
}

fn vulkano_format_to_skia(
    format: VulkanoFormat,
) -> Result<(skia_safe::gpu::vk::Format, skia_safe::ColorType), PlatformError> {
    match format {
        VulkanoFormat::B8G8R8A8_UNORM => {
            Ok((skia_safe::gpu::vk::Format::B8G8R8A8_UNORM, skia_safe::ColorType::BGRA8888))
        }
        VulkanoFormat::R8G8B8A8_UNORM => {
            Ok((skia_safe::gpu::vk::Format::R8G8B8A8_UNORM, skia_safe::ColorType::RGBA8888))
        }
        VulkanoFormat::B8G8R8A8_SRGB => {
            Ok((skia_safe::gpu::vk::Format::B8G8R8A8_SRGB, skia_safe::ColorType::BGRA8888))
        }
        VulkanoFormat::R8G8B8A8_SRGB => {
            Ok((skia_safe::gpu::vk::Format::R8G8B8A8_SRGB, skia_safe::ColorType::RGBA8888))
        }
        _ => Err(format!("Unsupported Vulkan format: {:?}", format).into()),
    }
}

fn vulkano_layout_to_skia(
    layout: VulkanoImageLayout,
) -> Result<skia_safe::gpu::vk::ImageLayout, PlatformError> {
    match layout {
        VulkanoImageLayout::Undefined => Ok(skia_safe::gpu::vk::ImageLayout::UNDEFINED),
        VulkanoImageLayout::General => Ok(skia_safe::gpu::vk::ImageLayout::GENERAL),
        VulkanoImageLayout::ColorAttachmentOptimal => {
            Ok(skia_safe::gpu::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        }
        VulkanoImageLayout::DepthStencilAttachmentOptimal => {
            Ok(skia_safe::gpu::vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        }
        VulkanoImageLayout::DepthStencilReadOnlyOptimal => {
            Ok(skia_safe::gpu::vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
        }
        VulkanoImageLayout::ShaderReadOnlyOptimal => {
            Ok(skia_safe::gpu::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        }
        VulkanoImageLayout::TransferSrcOptimal => {
            Ok(skia_safe::gpu::vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        }
        VulkanoImageLayout::TransferDstOptimal => {
            Ok(skia_safe::gpu::vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        }
        VulkanoImageLayout::Preinitialized => Ok(skia_safe::gpu::vk::ImageLayout::PREINITIALIZED),
        VulkanoImageLayout::PresentSrc => Ok(skia_safe::gpu::vk::ImageLayout::PRESENT_SRC_KHR),
        _ => Err(format!("Unsupported Vulkan image layout: {:?}", layout).into()),
    }
}

// Implement RendererSealed so this can be used as a Slint Renderer
impl RendererSealed for SkiaVulkanTextureRenderer {
    fn text_size(
        &self,
        text_item: Pin<&dyn i_slint_core::item_rendering::RenderString>,
        item_rc: &i_slint_core::items::ItemRc,
        max_width: Option<LogicalLength>,
        text_wrap: i_slint_core::items::TextWrap,
    ) -> LogicalSize {
        sharedparley::text_size(
            self,
            text_item,
            item_rc,
            max_width,
            text_wrap,
            Some(&self.text_layout_cache),
        )
        .unwrap_or_default()
    }

    fn char_size(
        &self,
        text_item: Pin<&dyn i_slint_core::item_rendering::HasFont>,
        item_rc: &i_slint_core::items::ItemRc,
        ch: char,
    ) -> LogicalSize {
        self.slint_context()
            .and_then(|ctx| {
                let mut font_ctx = ctx.font_context().borrow_mut();
                sharedparley::char_size(&mut font_ctx, text_item, item_rc, ch)
            })
            .unwrap_or_default()
    }

    fn font_metrics(
        &self,
        font_request: i_slint_core::graphics::FontRequest,
    ) -> i_slint_core::items::FontMetrics {
        self.slint_context()
            .map(|ctx| {
                let mut font_ctx = ctx.font_context().borrow_mut();
                sharedparley::font_metrics(&mut font_ctx, font_request)
            })
            .unwrap_or_default()
    }

    fn text_input_byte_offset_for_position(
        &self,
        text_input: Pin<&i_slint_core::items::TextInput>,
        item_rc: &i_slint_core::items::ItemRc,
        pos: LogicalPoint,
    ) -> usize {
        sharedparley::text_input_byte_offset_for_position(self, text_input, item_rc, pos)
    }

    fn text_input_cursor_rect_for_byte_offset(
        &self,
        text_input: Pin<&i_slint_core::items::TextInput>,
        item_rc: &i_slint_core::items::ItemRc,
        byte_offset: usize,
    ) -> LogicalRect {
        sharedparley::text_input_cursor_rect_for_byte_offset(self, text_input, item_rc, byte_offset)
    }

    fn register_font_from_memory(
        &self,
        data: &'static [u8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let ctx = self.slint_context().ok_or("slint platform not initialized")?;
        ctx.font_context().borrow_mut().collection.register_fonts(data.to_vec().into(), None);
        Ok(())
    }

    fn register_font_from_path(
        &self,
        path: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let requested_path = path.canonicalize().unwrap_or_else(|_| path.into());
        let contents = std::fs::read(requested_path)?;
        let ctx = self.slint_context().ok_or("slint platform not initialized")?;
        ctx.font_context().borrow_mut().collection.register_fonts(contents.into(), None);
        Ok(())
    }

    fn default_font_size(&self) -> LogicalLength {
        sharedparley::DEFAULT_FONT_SIZE
    }

    fn set_rendering_notifier(
        &self,
        _callback: Box<dyn i_slint_core::api::RenderingNotifier>,
    ) -> Result<(), i_slint_core::api::SetRenderingNotifierError> {
        // Rendering notifiers are not supported in texture rendering mode
        // because the rendering lifecycle is controlled externally
        Err(i_slint_core::api::SetRenderingNotifierError::Unsupported)
    }

    fn free_graphics_resources(
        &self,
        component: i_slint_core::item_tree::ItemTreeRef,
        _items: &mut dyn Iterator<Item = Pin<i_slint_core::items::ItemRef<'_>>>,
    ) -> Result<(), PlatformError> {
        self.image_cache.component_destroyed(component);
        self.path_cache.component_destroyed(component);
        self.text_layout_cache.component_destroyed(component);
        Ok(())
    }

    fn set_window_adapter(&self, window_adapter: &Rc<dyn WindowAdapter>) {
        *self.maybe_window_adapter.borrow_mut() = Some(Rc::downgrade(window_adapter));
        self.image_cache.clear_all();
        self.path_cache.clear_all();
        self.text_layout_cache.clear_all();
    }

    fn window_adapter(&self) -> Option<Rc<dyn WindowAdapter>> {
        self.maybe_window_adapter.borrow().as_ref().and_then(|w| w.upgrade())
    }

    fn resize(&self, size: i_slint_core::api::PhysicalSize) -> Result<(), PlatformError> {
        // Store the size for the next render
        *self.target_size.borrow_mut() = size;
        Ok(())
    }

    fn take_snapshot(
        &self,
    ) -> Result<
        i_slint_core::graphics::SharedPixelBuffer<i_slint_core::graphics::Rgba8Pixel>,
        PlatformError,
    > {
        Err("Snapshot not supported in texture rendering mode".into())
    }

    fn supports_transformations(&self) -> bool {
        true
    }
}
